import logging
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
def _get_valid_name(name):
    return name.replace('.', '_')