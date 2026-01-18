import random
import sys
from pathlib import Path
from typing import Any, Optional
import fastai
from fastai.callbacks import TrackerCallback
import wandb
def _wandb_log_predictions(self) -> None:
    """Log prediction samples."""
    pred_log = []
    if self.validation_data is None:
        return
    for x, y in self.validation_data:
        try:
            pred = self.learn.predict(x)
        except Exception:
            raise FastaiError('Unable to run "predict" method from Learner to log prediction samples.')
        if not pred[1].shape or pred[1].dim() == 1:
            pred_log.append(wandb.Image(x.data, caption=f'Ground Truth: {y}\nPrediction: {pred[0]}'))
        elif hasattr(x, 'show'):
            pred_log.append(wandb.Image(x.data, caption='Input data', grouping=3))
            for im, capt in ((pred[0], 'Prediction'), (y, 'Ground Truth')):
                my_dpi = 100
                fig = plt.figure(frameon=False, dpi=my_dpi)
                h, w = x.size
                fig.set_size_inches(w / my_dpi, h / my_dpi)
                ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                ax.set_axis_off()
                fig.add_axes(ax)
                x.show(ax=ax, y=im)
                pred_log.append(wandb.Image(fig, caption=capt))
                plt.close(fig)
        elif hasattr(y, 'shape') and (len(y.shape) == 2 or (len(y.shape) == 3 and y.shape[0] in [1, 3, 4])):
            pred_log.extend([wandb.Image(x.data, caption='Input data', grouping=3), wandb.Image(pred[0].data, caption='Prediction'), wandb.Image(y.data, caption='Ground Truth')])
        else:
            pred_log.append(wandb.Image(x.data, caption='Input data'))
        wandb.log({'Prediction Samples': pred_log}, commit=False)