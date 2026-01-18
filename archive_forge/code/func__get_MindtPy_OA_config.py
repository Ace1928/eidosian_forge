import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _get_MindtPy_OA_config():
    """Set up the configurations for MindtPy-OA.

    Returns
    -------
    CONFIG : ConfigBlock
        The specific configurations for MindtPy
    """
    CONFIG = ConfigBlock('MindtPy-OA')
    _add_common_configs(CONFIG)
    _add_oa_configs(CONFIG)
    _add_roa_configs(CONFIG)
    _add_fp_configs(CONFIG)
    _add_oa_cuts_configs(CONFIG)
    _add_subsolver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    _add_bound_configs(CONFIG)
    return CONFIG