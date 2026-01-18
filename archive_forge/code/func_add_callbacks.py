from typing import Any, Callable, Dict, List, Optional
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
import wandb
from wandb.sdk.lib import telemetry
def add_callbacks(yolo: YOLO, run_name: Optional[str]=None, project: Optional[str]=None, tags: Optional[List[str]]=None, resume: Optional[str]=None, **kwargs: Optional[Any]) -> YOLO:
    """A YOLO model wrapper that tracks metrics, and logs models to Weights & Biases.

    Args:
        yolo: A YOLOv8 model that's inherited from `:class:ultralytics.yolo.engine.model.YOLO`
        run_name, str: The name of the Weights & Biases run, defaults to an auto generated name if `trainer.args.name` is not defined.
        project, str: The name of the Weights & Biases project, defaults to `"YOLOv8"` if `trainer.args.project` is not defined.
        tags, List[str]: A list of tags to be added to the Weights & Biases run, defaults to `["YOLOv8"]`.
        resume, str: Whether to resume a previous run on Weights & Biases, defaults to `None`.
        **kwargs: Additional arguments to be passed to `wandb.init()`.

    Usage:
    ```python
    from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks

    model = YOLO("yolov8n.pt")
    add_wandb_callbacks(
        model,
    )
    model.train(
        data="coco128.yaml",
        epochs=3,
        imgsz=640,
    )
    ```
    """
    wandb.termwarn('The wandb callback is currently in beta and is subject to change based on updates to `ultralytics yolov8`.\n        The callback is tested and supported for ultralytics v8.0.43 and above.\n        Please report any issues to https://github.com/wandb/wandb/issues with the tag `yolov8`.\n        ', repeat=False)
    wandb.termwarn('This wandb callback is no longer functional and would be deprecated in the near future.\n        We recommend you to use the updated callback using `from wandb.integration.ultralytics import add_wandb_callback`.\n        The updated callback is tested and supported for ultralytics 8.0.167 and above.\n        You can refer to https://docs.wandb.ai/guides/integrations/ultralytics for the updated documentation.\n        Please report any issues to https://github.com/wandb/wandb/issues with the tag `yolov8`.\n        ', repeat=False)
    if RANK in [-1, 0]:
        wandb_logger = WandbCallback(yolo, run_name=run_name, project=project, tags=tags, resume=resume, **kwargs)
        for event, callback_fn in wandb_logger.callbacks.items():
            yolo.add_callback(event, callback_fn)
        return yolo
    else:
        wandb.termerror('The RANK of the process to add the callbacks was neither 0 or -1.No Weights & Biases callbacks were added to this instance of the YOLO model.')
    return yolo