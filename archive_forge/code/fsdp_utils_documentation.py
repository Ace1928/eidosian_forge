from torch.distributed._tensor.device_mesh import init_device_mesh
import os 

     Initializes a device mesh for use with Hybrid Sharding strategy in FSDP (HSDP) training.

    This function requires explicit sizes for replica and sharding groups to accommodate models
    whose GPU fit is unknown, providing flexibility in distributed training setups.
    
    Args:
        replica_group_size (int): The size of each replica group. Must be provided to ensure
            the model fits within the available resources.
        sharding_group_size (int): The size of each sharding group that the model can fit. Must be provided to 
            ensure the correct distribution of model parameters.
        device (str, optional): The device to use (e.g., "cuda:0"). If None, defaults to "cuda"
            with the local rank as the device index.

    Returns:
        A device mesh object compatible with FSDP.

    Raises:
        ValueError: If replica_group_size or sharding_group_size are not provided, or if the
            world size is not evenly divisible by the sharding group size.
        RuntimeError: If a valid device mesh cannot be created.

    Usage:
        If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups
        >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
        >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    