import os
Analyzes the given tflite_model with dumping model structure.

    This tool provides a way to understand users' TFLite flatbuffer model by
    dumping internal graph structure. It also provides additional features
    like checking GPU delegate compatibility.

    WARNING: Experimental interface, subject to change.
             The output format is not guaranteed to stay stable, so don't
             write scripts to this.

    Args:
      model_path: TFLite flatbuffer model path.
      model_content: TFLite flatbuffer model object.
      gpu_compatibility: Whether to check GPU delegate compatibility.
      **kwargs: Experimental keyword arguments to analyze API.

    Returns:
      Print analyzed report via console output.
    