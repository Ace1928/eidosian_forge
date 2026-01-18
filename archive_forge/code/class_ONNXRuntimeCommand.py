from .. import BaseOptimumCLICommand, CommandInfo
from .optimize import ONNXRuntimeOptimizeCommand
from .quantize import ONNXRuntimeQuantizeCommand
class ONNXRuntimeCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name='onnxruntime', help='ONNX Runtime optimize and quantize utilities.')
    SUBCOMMANDS = (CommandInfo(name='optimize', help='Optimize ONNX models.', subcommand_class=ONNXRuntimeOptimizeCommand), CommandInfo(name='quantize', help='Dynammic quantization for ONNX models.', subcommand_class=ONNXRuntimeQuantizeCommand))