import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
class MegatronLMDummyDataLoader:
    """
    Dummy dataloader presents model parameters or param groups, this is primarily used to follow conventional training

    Args:
        **dataset_kwargs: Megatron data arguments.
    """

    def __init__(self, **dataset_kwargs):
        parser = argparse.ArgumentParser()
        parser = _add_data_args(parser)
        parser = _add_validation_args(parser)
        data_args = parser.parse_known_args()
        self.dataset_args = vars(data_args[0])
        self.dataset_args.update(dataset_kwargs)
        self.dataset_args['megatron_dataset_flag'] = True

    def set_megatron_data_args(self):
        args = get_args()
        for key, value in self.dataset_args.items():
            setattr(args, key, value)

    def get_train_valid_test_datasets_provider(self):

        def train_valid_test_datasets_provider(train_val_test_num_samples):
            """Build train, valid, and test datasets."""
            args = get_args()
            dataset_args = {'data_prefix': args.data_path, 'data_impl': args.data_impl, 'splits_string': args.split, 'train_valid_test_num_samples': train_val_test_num_samples, 'skip_warmup': not args.mmap_warmup, 'seed': args.seed}
            if args.model_type_name == 'bert':
                dataset_args.update({'max_seq_length': args.seq_length, 'masked_lm_prob': args.mask_prob, 'short_seq_prob': args.short_seq_prob, 'binary_head': args.bert_binary_head})
            elif args.model_type_name == 'gpt':
                dataset_args.update({'seq_length': args.seq_length})
            elif args.model_type_name == 't5':
                dataset_args.update({'max_seq_length': args.encoder_seq_length, 'max_seq_length_dec': args.decoder_seq_length, 'masked_lm_prob': args.mask_prob, 'short_seq_prob': args.short_seq_prob, 'dataset_type': 't5'})
            else:
                raise ValueError(f'Unsupported model type: {args.model_type_name}')
            if args.model_type_name == 'gpt':
                from megatron.data.gpt_dataset import build_train_valid_test_datasets
            else:
                from megatron.data.dataset_utils import build_train_valid_test_datasets
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(**dataset_args)
            return (train_ds, valid_ds, test_ds)
        return train_valid_test_datasets_provider

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        if dataset is None:
            return None
        args = get_args()
        micro_batch_size = args.micro_batch_size * args.num_micro_batches
        if args.dataloader_type == 'single':
            batch_sampler = MegatronPretrainingSampler(total_samples=len(dataset), consumed_samples=consumed_samples, micro_batch_size=micro_batch_size, data_parallel_rank=mpu.get_data_parallel_rank(), data_parallel_size=mpu.get_data_parallel_world_size())
        elif args.dataloader_type == 'cyclic':
            batch_sampler = MegatronPretrainingRandomSampler(dataset, total_samples=len(dataset), consumed_samples=consumed_samples, micro_batch_size=micro_batch_size, data_parallel_rank=mpu.get_data_parallel_rank(), data_parallel_size=mpu.get_data_parallel_world_size(), data_sharding=args.data_sharding)
        else:
            raise Exception(f'{args.dataloader_type} dataloader type is not supported.')
        return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

    def build_train_valid_test_data_iterators(self):

        def cyclic_iter(iter):
            while True:
                yield from iter
        args = get_args()
        train_dataloader, valid_dataloader, test_dataloader = (None, None, None)
        print_rank_0('> building train, validation, and test datasets ...')
        if args.iteration > 0 and args.consumed_train_samples == 0:
            assert args.train_samples is None, 'only backward compatiblity support for iteration-based training'
            args.consumed_train_samples = args.iteration * args.global_batch_size
        if args.iteration > 0 and args.consumed_valid_samples == 0:
            if args.train_samples is None:
                args.consumed_valid_samples = args.iteration // args.eval_interval * args.eval_iters * args.global_batch_size
        if mpu.get_tensor_model_parallel_rank() == 0:
            if args.train_samples:
                train_samples = args.train_samples
            else:
                train_samples = args.train_iters * args.global_batch_size
            eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
            test_iters = args.eval_iters
            train_val_test_num_samples = [train_samples, eval_iters * args.global_batch_size, test_iters * args.global_batch_size]
            print_rank_0(' > datasets target sizes (minimum size):')
            print_rank_0(f'    train:      {train_val_test_num_samples[0]}')
            print_rank_0(f'    validation: {train_val_test_num_samples[1]}')
            print_rank_0(f'    test:       {train_val_test_num_samples[2]}')
            train_valid_test_datasets_provider = self.get_train_valid_test_datasets_provider()
            train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(train_val_test_num_samples)
            train_dataloader = self.build_pretraining_data_loader(train_ds, args.consumed_train_samples)
            valid_dataloader = self.build_pretraining_data_loader(valid_ds, args.consumed_valid_samples)
            test_dataloader = self.build_pretraining_data_loader(test_ds, 0)
            do_train = train_dataloader is not None and args.train_iters > 0
            do_valid = valid_dataloader is not None and args.eval_iters > 0
            do_test = test_dataloader is not None and args.eval_iters > 0
            flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
        else:
            flags = torch.cuda.LongTensor([0, 0, 0])
        torch.distributed.broadcast(flags, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
        args.do_train = flags[0].item()
        args.do_valid = flags[1].item()
        args.do_test = flags[2].item()
        dl_type = args.dataloader_type
        assert dl_type in ['single', 'cyclic']
        if train_dataloader is not None:
            train_data_iterator = iter(train_dataloader) if dl_type == 'single' else iter(cyclic_iter(train_dataloader))
        else:
            train_data_iterator = None
        if valid_dataloader is not None:
            valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' else iter(cyclic_iter(valid_dataloader))
        else:
            valid_data_iterator = None
        if test_dataloader is not None:
            test_data_iterator = iter(test_dataloader) if dl_type == 'single' else iter(cyclic_iter(test_dataloader))
        else:
            test_data_iterator = None
        return (train_data_iterator, valid_data_iterator, test_data_iterator)