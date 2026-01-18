import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
class HardSwish(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node('HardSwish', inputs=['x'], outputs=['y'])
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = hardswish(x)
        expect(node, inputs=[x], outputs=[y], name='test_hardswish')