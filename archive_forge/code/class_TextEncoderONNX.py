from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
class TextEncoderONNX:

    def __init__(self, text_encoder_path: str, reranker_path: str, device: str):
        """
        :param text_encoder_path: Path to onnx of text encoder
        :param reranker_path: Path to onnx of reranker
        :param device: Device name, either cpu or gpu
        """
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.text_encoder_session = ort.InferenceSession(text_encoder_path, sess_options=session_options, providers=available_providers(device))
        self.reranker_session = ort.InferenceSession(reranker_path, sess_options=session_options, providers=available_providers(device))

    def __call__(self, input_ids: ndarray, attention_mask: ndarray) -> Tuple[ndarray, ndarray]:
        return self.text_encoder_session.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})

    def forward_multimodal(self, text_features: ndarray, attention_mask: ndarray, image_features: ndarray) -> Tuple[ndarray, ndarray]:
        return self.reranker_session.run(None, {'text_features': text_features, 'attention_mask': attention_mask, 'image_features': image_features})