from parlai.core.build_data import modelzoo_path
from parlai.core.loader import load_agent_module
from parlai.core.loader import register_agent  # noqa: F401
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
import copy
import os
import parlai.utils.logging as logging
def add_datapath_and_model_args(opt: Opt):
    from parlai.core.params import ParlaiParser, get_model_name
    parser = ParlaiParser(add_parlai_args=False)
    parser.add_parlai_data_path()
    model = get_model_name(opt)
    if model is not None:
        parser.add_model_subargs(model)
    opt_parser = parser.parse_args('')
    for k, v in opt_parser.items():
        if k not in opt:
            opt[k] = v