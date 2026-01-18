import torch.distributed as dist
import torch.distributed.rpc as rpc
def _faulty_tensorpipe_init_backend_handler(store, name, rank, world_size, rpc_backend_options):
    from . import FaultyTensorPipeAgent
    from . import FaultyTensorPipeRpcBackendOptions
    from torch.distributed.rpc import api
    if not isinstance(store, dist.Store):
        raise TypeError(f'`store` must be a c10d::Store. {store}')
    if not isinstance(rpc_backend_options, FaultyTensorPipeRpcBackendOptions):
        raise TypeError(f'`rpc_backend_options` must be a `FaultyTensorPipeRpcBackendOptions`. {rpc_backend_options}')
    agent = FaultyTensorPipeAgent(store, name, rank, world_size, rpc_backend_options, {}, [])
    api._init_rpc_states(agent)
    return agent