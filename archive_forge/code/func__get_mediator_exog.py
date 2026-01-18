import numpy as np
import pandas as pd
from statsmodels.graphics.utils import maybe_name_or_idx
def _get_mediator_exog(self, exposure):
    """
        Return the mediator exog matrix with exposure set to the given
        value.  Set values of moderated variables as needed.
        """
    mediator_exog = self._mediator_exog
    if not hasattr(self.mediator_model, 'formula'):
        mediator_exog[:, self._exp_pos_mediator] = exposure
        for ix in self.moderators:
            v = self.moderators[ix]
            mediator_exog[:, ix[1]] = v
    else:
        df = self.mediator_model.data.frame.copy()
        df[self.exposure] = exposure
        for vname in self.moderators:
            v = self.moderators[vname]
            df.loc[:, vname] = v
        klass = self.mediator_model.__class__
        init_kwargs = self.mediator_model._get_init_kwds()
        model = klass.from_formula(data=df, **init_kwargs)
        mediator_exog = model.exog
    return mediator_exog