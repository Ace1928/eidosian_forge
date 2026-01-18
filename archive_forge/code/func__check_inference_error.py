import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def _check_inference_error(self, model_text: str):
    """Check that the model inference raises an InferenceError."""
    model = onnx.parser.parse_model(model_text)
    with self.assertRaises(onnx.shape_inference.InferenceError):
        onnx.shape_inference.infer_shapes(model, True, True)