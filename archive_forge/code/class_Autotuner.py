from __future__ import annotations
import builtins
import time
from typing import Dict
from ..testing import do_bench
from .jit import KernelInterface
class Autotuner(KernelInterface):

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, prune_configs_by: Dict=None, warmup=25, rep=100):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=2, num_ctas=1)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = {}
        self.arg_names = arg_names
        self.reset_idx = []
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]
        self.restore_idx = []
        if restore_value is not None:
            self.restore_idx = [arg_names.index(k) for k in restore_value]
        self.pre_hook = lambda args, reset_only=False: 0
        self.post_hook = lambda args: 0
        if len(self.reset_idx) > 0 or len(self.restore_idx) > 0:

            def _pre_hook(args, reset_only=False):
                for i in self.reset_idx:
                    args[i].zero_()
                if not reset_only:
                    self.restore_copies = [args[i].clone() for i in self.restore_idx]
            self.pre_hook = _pre_hook
        if len(self.restore_idx) > 0:

            def _post_hook(args):
                for i, j in enumerate(self.restore_idx):
                    args[j].copy_(self.restore_copies[i])
                self.restore_copies = []
            self.post_hook = _post_hook
        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get('perf_model', self.perf_model)
            self.configs_top_k = prune_configs_by.get('top_k', self.configs_top_k)
            self.early_config_prune = prune_configs_by.get('early_config_prune', self.early_config_prune)
        self.fn = fn
        self.warmup = warmup
        self.rep = rep

    def _bench(self, *args, config, **meta):
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}. Make sure that you don't re-define auto-tuned symbols.")
        current = dict(meta, **config.kwargs)
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, num_ctas=config.num_ctas, enable_warp_specialization=config.enable_warp_specialization, **current)
            self.post_hook(args)
        try:
            return do_bench(kernel_call, warmup=self.warmup, rep=self.rep, quantiles=(0.5, 0.2, 0.8))
        except OutOfResources:
            return [float('inf'), float('inf'), float('inf')]

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = []
            for name in self.arg_names:
                if name in all_args:
                    _args.append(all_args[name])
            key = [_args[i] for i in self.key_idx]
            for arg in _args:
                if hasattr(arg, 'dtype'):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.pre_hook(args, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        full_nargs = {**self.nargs, **kwargs, **self.best_config.kwargs}
        if config.pre_hook is not None:
            config.pre_hook(full_nargs)
        ret = self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, num_ctas=config.num_ctas, enable_warp_specialization=config.enable_warp_specialization, **kwargs, **config.kwargs)
        self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {config: self.perf_model(**self.nargs, **kwargs, **config.kwargs, num_stages=config.num_stages, num_warps=config.num_warps, num_ctas=config.num_ctas, enable_warp_specialization=config.enable_warp_specialization, enable_persistent=config.enable_persistent) for config in pruned_configs}
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        for config in self.prune_configs(kwargs):
            self.fn.warmup(*args, num_warps=config.num_warps, num_ctas=config.num_ctas, num_stages=config.num_stages, enable_warp_specialization=config.enable_warp_specialization, enable_persistent=config.enable_persistent, **kwargs, **config.kwargs)
        self.nargs = None