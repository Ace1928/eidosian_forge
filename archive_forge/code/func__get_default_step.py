import logging
from typing import Any, Optional
from mlflow.recipes.recipe import BaseRecipe
from mlflow.recipes.step import BaseStep
from mlflow.recipes.steps.evaluate import EvaluateStep
from mlflow.recipes.steps.ingest import IngestScoringStep, IngestStep
from mlflow.recipes.steps.predict import PredictStep
from mlflow.recipes.steps.register import RegisterStep
from mlflow.recipes.steps.split import SplitStep
from mlflow.recipes.steps.train import TrainStep
from mlflow.recipes.steps.transform import TransformStep
def _get_default_step(self) -> BaseStep:
    return self._steps[self._DEFAULT_STEP_INDEX]