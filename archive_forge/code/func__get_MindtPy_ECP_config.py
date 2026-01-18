import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _get_MindtPy_ECP_config():
    """Set up the configurations for MindtPy-ECP.

    Returns
    -------
    CONFIG : ConfigBlock
        The specific configurations for MindtPy
    """
    CONFIG = ConfigBlock('MindtPy-GOA')
    _add_common_configs(CONFIG)
    _add_ecp_configs(CONFIG)
    _add_oa_cuts_configs(CONFIG)
    _add_subsolver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    _add_bound_configs(CONFIG)
    return CONFIG