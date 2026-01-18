from typing import Union
from ..data import from_cmdstanpy, from_pystan
from .base import SamplingWrapper
class PyStan2SamplingWrapper(StanSamplingWrapper):
    """PyStan (2.x) sampling wrapper base class.

    See the documentation on  :class:`~arviz.SamplingWrapper` for a more detailed
    description. An example of ``PyStanSamplingWrapper`` usage can be found
    in the :ref:`pystan_refitting` notebook. For usage examples of other wrappers
    see the user guide pages on :ref:`wrapper_guide`.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.

    See Also
    --------
    SamplingWrapper
    """

    def sample(self, modified_observed_data):
        """Resample the PyStan model stored in self.model on modified_observed_data."""
        fit = self.model.sampling(data=modified_observed_data, **self.sample_kwargs)
        return fit