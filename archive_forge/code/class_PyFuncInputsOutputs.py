from abc import abstractmethod
from dataclasses import dataclass
from typing import List
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.models.utils import PyFuncInput, PyFuncOutput
@dataclass
class PyFuncInputsOutputs:
    inputs: List[PyFuncInput]
    outputs: List[PyFuncOutput] = None