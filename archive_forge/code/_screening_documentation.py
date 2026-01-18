from collections import defaultdict
import numpy as np
from statsmodels.base._penalties import SCADSmoothed

        batched version of screen exog

        This screens variables in a two step process:

        In the first step screen_exog is used on each element of the
        exog_iterator, and the batch winners are collected.

        In the second step all batch winners are combined into a new array
        of exog candidates and `screen_exog` is used to select a final
        model.

        Parameters
        ----------
        exog_iterator : iterator over ndarrays

        Returns
        -------
        res_screen_final : instance of ScreeningResults
            This is the instance returned by the second round call to
            `screen_exog`. Additional attributes are added to provide
            more information about the batched selection process.
            The index of final nonzero variables is
            `idx_nonzero_batches` which is a 2-dimensional array with batch
            index in the first column and variable index within batch in the
            second column. They can be used jointly as index for the data
            in the exog_iterator.
            see ScreeningResults for a full description
        