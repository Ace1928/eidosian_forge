from typing import List, Optional
import torch
from torch.backends._nnapi.serializer import _NnapiSerializer
class NnapiModule(torch.nn.Module):
    """Torch Module that wraps an NNAPI Compilation.

    This module handles preparing the weights, initializing the
    NNAPI TorchBind object, and adjusting the memory formats
    of all inputs and outputs.
    """
    comp: Optional[torch.classes._nnapi.Compilation]
    weights: List[torch.Tensor]
    out_templates: List[torch.Tensor]

    def __init__(self, shape_compute_module: torch.nn.Module, ser_model: torch.Tensor, weights: List[torch.Tensor], inp_mem_fmts: List[int], out_mem_fmts: List[int], compilation_preference: int, relax_f32_to_f16: bool):
        super().__init__()
        self.shape_compute_module = shape_compute_module
        self.ser_model = ser_model
        self.weights = weights
        self.inp_mem_fmts = inp_mem_fmts
        self.out_mem_fmts = out_mem_fmts
        self.out_templates = []
        self.comp = None
        self.compilation_preference = compilation_preference
        self.relax_f32_to_f16 = relax_f32_to_f16

    @torch.jit.export
    def init(self, args: List[torch.Tensor]):
        assert self.comp is None
        self.out_templates = self.shape_compute_module.prepare(self.ser_model, args)
        self.weights = [w.contiguous() for w in self.weights]
        comp = torch.classes._nnapi.Compilation()
        comp.init2(self.ser_model, self.weights, self.compilation_preference, self.relax_f32_to_f16)
        self.comp = comp

    def forward(self, args: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.comp is None:
            self.init(args)
        comp = self.comp
        assert comp is not None
        outs = [torch.empty_like(out) for out in self.out_templates]
        assert len(args) == len(self.inp_mem_fmts)
        fixed_args = []
        for idx in range(len(args)):
            fmt = self.inp_mem_fmts[idx]
            if fmt == 0:
                fixed_args.append(args[idx].contiguous())
            elif fmt == 1:
                fixed_args.append(args[idx].permute(0, 2, 3, 1).contiguous())
            else:
                raise Exception('Invalid mem_fmt')
        comp.run(fixed_args, outs)
        assert len(outs) == len(self.out_mem_fmts)
        for idx in range(len(self.out_templates)):
            fmt = self.out_mem_fmts[idx]
            if fmt in (0, 2):
                pass
            elif fmt == 1:
                outs[idx] = outs[idx].permute(0, 3, 1, 2)
            else:
                raise Exception('Invalid mem_fmt')
        return outs