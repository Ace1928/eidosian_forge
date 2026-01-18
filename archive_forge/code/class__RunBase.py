import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
@dataclass
class _RunBase:
    model_name_or_path: str = field(metadata={'description': 'Name of the model hosted on the Hub to use for the run.'})
    task: str = field(metadata={'description': 'Task performed by the model.'})
    quantization_approach: QuantizationApproach = field(metadata={'description': 'Whether to use dynamic or static quantization.'})
    dataset: DatasetArgs = field(metadata={'description': 'Dataset to use. Several keys must be set on top of the dataset name.'})
    framework: Frameworks = field(metadata={'description': 'Name of the framework used (e.g. "onnxruntime").'})
    framework_args: FrameworkArgs = field(metadata={'description': 'Framework-specific arguments.'})