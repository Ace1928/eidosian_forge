import torch
import torch.nn.functional as F
def _move_exported_model_to_eval(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to eval mode.

    This is equivalent to model.eval() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing inference on the model.
    """
    _replace_dropout_for_eval(model)
    _replace_batchnorm_for_eval(model)
    return model