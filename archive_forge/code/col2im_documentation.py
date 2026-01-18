import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
Col2Im operator with N-dimension support

    The tests below can be reproduced in Python using https://github.com/f-dangel/unfoldNd/
    