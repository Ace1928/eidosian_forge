from pathlib import Path
from typing import TYPE_CHECKING
from optimum.commands.base import BaseOptimumCLICommand
class ONNXRuntimeOptimizeCommand(BaseOptimumCLICommand):

    @staticmethod
    def parse_args(parser: 'ArgumentParser'):
        return parse_args_onnxruntime_optimize(parser)

    def run(self):
        from ...onnxruntime.configuration import AutoOptimizationConfig, ORTConfig
        from ...onnxruntime.optimization import ORTOptimizer
        if self.args.output == self.args.onnx_model:
            raise ValueError('The output directory must be different than the directory hosting the ONNX model.')
        save_dir = self.args.output
        file_names = [model.name for model in self.args.onnx_model.glob('*.onnx')]
        optimizer = ORTOptimizer.from_pretrained(self.args.onnx_model, file_names)
        if self.args.config:
            optimization_config = ORTConfig
        elif self.args.O1:
            optimization_config = AutoOptimizationConfig.O1()
        elif self.args.O2:
            optimization_config = AutoOptimizationConfig.O2()
        elif self.args.O3:
            optimization_config = AutoOptimizationConfig.O3()
        elif self.args.O4:
            optimization_config = AutoOptimizationConfig.O4()
        else:
            optimization_config = ORTConfig.from_pretained(self.args.config).optimization
        optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)