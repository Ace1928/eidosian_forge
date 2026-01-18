from ...utils import split_and_load
from .... import autograd
Trains the estimator model on a batch of training data.

        Parameters
        ----------
        estimator : Estimator
            Reference to the estimator
        train_batch : tuple
            Data and label of a batch from the training data loader.
        batch_axis : int, default 0
            Batch axis to split the training data into devices.

        Returns
        -------
        data: List of NDArray
            Sharded data from the batch. Data is sharded with
            `gluon.split_and_load`.
        label: List of NDArray
            Sharded label from the batch. Labels are sharded with
            `gluon.split_and_load`.
        pred: List of NDArray
            Prediction on each of the sharded inputs.
        loss: List of NDArray
            Loss on each of the sharded inputs.
        