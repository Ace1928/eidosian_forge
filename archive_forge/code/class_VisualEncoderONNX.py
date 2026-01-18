from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
class VisualEncoderONNX:

    def __init__(self, model_path: str, device: str):
        """
        :param model_path: Path to onnx model
        :param device: Device name, either cpu or gpu
        """
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=available_providers(device))

    def __call__(self, images: ndarray) -> Tuple[ndarray, ndarray]:
        return self.session.run(None, {'images': images})