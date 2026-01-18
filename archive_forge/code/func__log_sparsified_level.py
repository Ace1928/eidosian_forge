import logging
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
def _log_sparsified_level(model, data_sparsifier) -> None:
    for name, parameter in model.named_parameters():
        if type(parameter) not in SUPPORTED_TYPES:
            continue
        valid_name = _get_valid_name(name)
        mask = data_sparsifier.get_mask(name=valid_name)
        sparsity_level = 1.0 - mask.float().mean()
        logger.info('Sparsity in layer %s = % .2%', name, sparsity_level)