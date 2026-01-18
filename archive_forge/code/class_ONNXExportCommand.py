import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING
from ...exporters import TasksManager
from ...utils import DEFAULT_DUMMY_SHAPES
from ..base import BaseOptimumCLICommand
class ONNXExportCommand(BaseOptimumCLICommand):

    @staticmethod
    def parse_args(parser: 'ArgumentParser'):
        return parse_args_onnx(parser)

    def run(self):
        from ...exporters.onnx import main_export
        input_shapes = {}
        for input_name in DEFAULT_DUMMY_SHAPES.keys():
            if hasattr(self.args, input_name):
                input_shapes[input_name] = getattr(self.args, input_name)
        main_export(model_name_or_path=self.args.model, output=self.args.output, task=self.args.task, opset=self.args.opset, device=self.args.device, fp16=self.args.fp16, dtype=self.args.dtype, optimize=self.args.optimize, monolith=self.args.monolith, no_post_process=self.args.no_post_process, framework=self.args.framework, atol=self.args.atol, cache_dir=self.args.cache_dir, trust_remote_code=self.args.trust_remote_code, pad_token_id=self.args.pad_token_id, for_ort=self.args.for_ort, use_subprocess=True, _variant=self.args.variant, library_name=self.args.library_name, legacy=self.args.legacy, no_dynamic_axes=self.args.no_dynamic_axes, model_kwargs=self.args.model_kwargs, do_constant_folding=not self.args.no_constant_folding, **input_shapes)