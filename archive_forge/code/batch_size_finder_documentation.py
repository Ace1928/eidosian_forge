from typing import Optional
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.tuner.batch_size_scaling import _scale_batch_size
from pytorch_lightning.utilities.exceptions import MisconfigurationException, _TunerExitException
from pytorch_lightning.utilities.parsing import lightning_hasattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
Finds the largest batch size supported by a given model before encountering an out of memory (OOM) error.

    All you need to do is add it as a callback inside Trainer and call ``trainer.{fit,validate,test,predict}``.
    Internally, it calls the respective step function ``steps_per_trial`` times for each batch size until one
    of the batch sizes generates an OOM error.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        mode: search strategy to update the batch size:

            - ``'power'``: Keep multiplying the batch size by 2, until we get an OOM error.
            - ``'binsearch'``: Initially keep multiplying by 2 and after encountering an OOM error
              do a binary search between the last successful batch size and the batch size that failed.

        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs,
            however in practice a few are needed.

        init_val: initial batch size to start the search with.

        max_trials: max number of increases in batch size done before
            algorithm is terminated

        batch_arg_name: name of the attribute that stores the batch size.
            It is expected that the user has provided a model or datamodule that has a hyperparameter
            with that name. We will look for this attribute name in the following places

            - ``model``
            - ``model.hparams``
            - ``trainer.datamodule`` (the datamodule passed to the tune method)

    Example::

        # 1. Customize the BatchSizeFinder callback to run at different epochs. This feature is
        # useful while fine-tuning models since you can't always use the same batch size after
        # unfreezing the backbone.
        from pytorch_lightning.callbacks import BatchSizeFinder


        class FineTuneBatchSizeFinder(BatchSizeFinder):
            def __init__(self, milestones, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.milestones = milestones

            def on_fit_start(self, *args, **kwargs):
                return

            def on_train_epoch_start(self, trainer, pl_module):
                if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                    self.scale_batch_size(trainer, pl_module)


        trainer = Trainer(callbacks=[FineTuneBatchSizeFinder(milestones=(5, 10))])
        trainer.fit(...)

    Example::

        # 2. Run batch size finder for validate/test/predict.
        from pytorch_lightning.callbacks import BatchSizeFinder


        class EvalBatchSizeFinder(BatchSizeFinder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def on_fit_start(self, *args, **kwargs):
                return

            def on_test_start(self, trainer, pl_module):
                self.scale_batch_size(trainer, pl_module)


        trainer = Trainer(callbacks=[EvalBatchSizeFinder()])
        trainer.test(...)

    