import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from .model import LORA_CONFIG
class train_callback(pl.Callback):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))
            if args.lr_final == 0 or args.lr_init == 0:
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
        if args.my_exit_tokens != 0:
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init
            lr_mult = 0.5 + lr_final_factor / 2 + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if trainer.is_global_zero or 'deepspeed_stage_3' in args.strategy:
                    my_save(args, trainer, pl_module.state_dict(), f'{args.proj_dir}/rwkv-final.pth')
                    exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)
        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay
        for param_group in trainer.optimizers[0].param_groups:
            if param_group['weight_decay'] > 0:
                param_group['weight_decay'] = wd_now
            if args.layerwise_lr > 0:
                param_group['lr'] = lr * param_group['my_lr_scale']
            else:
                param_group['lr'] = lr
        trainer.my_lr = lr
        trainer.my_wd = wd_now
        if trainer.global_step == 0:
            if trainer.is_global_zero:
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + '/train_log.txt', 'a')
                trainer.my_log.write(f'NEW RUN {args.my_timestamp}\n{vars(self.args)}\n')
                try:
                    print(f'\n{trainer.strategy.config}\n')
                    trainer.my_log.write(f'{trainer.strategy.config}\n')
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print('Login to wandb...')
                    import wandb
                    wandb.init(project=args.wandb, name=args.run_name + ' ' + args.my_timestamp, config=args, save_code=False)
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1000000000.0
                kt_s = token_per_step / t_cost / 1000
                self.log('REAL it/s', 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log('Kt/s', kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0] == '2':
                trainer.my_loss = outputs['loss']
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log('lr', trainer.my_lr, prog_bar=True, on_step=True)
            self.log('loss', trainer.my_epoch_loss, prog_bar=True, on_step=True)
            if len(args.wandb) > 0:
                lll = {'loss': trainer.my_loss, 'lr': trainer.my_lr, 'wd': trainer.my_wd, 'Gtokens': real_step * token_per_step / 1000000000.0}
                if kt_s > 0:
                    lll['kt/s'] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))
        if trainer.is_global_zero or 'deepspeed_stage_3' in args.strategy:
            if args.magic_prime > 0:
                expand_factor = 2 if args.my_qa_mask > 0 else 1
                if int(real_step) == int(args.magic_prime * expand_factor // args.real_bsz) - 1 + int(args.my_random_steps):
                    to_save_dict = pl_module.state_dict()
                    my_save(args, trainer, to_save_dict, f'{args.proj_dir}/rwkv-final.pth')

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0] == '2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        assert 'MyDataset' in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if trainer.is_global_zero or 'deepspeed_stage_3' in args.strategy:
            if args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0 or trainer.current_epoch == args.epoch_count - 1:
                if args.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                if args.data_type == 'img' and (not args.lora):
                    for name, state in to_save_dict.items():
                        if 'img' in name:
                            to_save_dict[name] = state
                if args.lora:
                    enable_time_finetune = 'time' in LORA_CONFIG['parts']
                    enable_ln_finetune = 'ln' in LORA_CONFIG['parts']
                    lora_dict = {}
                    for name, state in to_save_dict.items():
                        if 'img' in name:
                            lora_dict[name] = state
                        if '.lora_' in name or (enable_time_finetune and '.time_' in name) or (enable_ln_finetune and '.ln' in name):
                            lora_dict[name] = state
                    to_save_dict = lora_dict
                try:
                    my_save(args, trainer, to_save_dict, f'{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth')
                except Exception as e:
                    print('Error\n\n', e, '\n\n')
        if trainer.is_global_zero:
            trainer.my_log.write(f'{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n')
            trainer.my_log.flush()
            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if args.epoch_begin + trainer.current_epoch >= args.my_exit:
                exit(0)