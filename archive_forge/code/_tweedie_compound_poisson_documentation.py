import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln

Private experimental module for miscellaneous Tweedie functions.

References
----------

Dunn, Peter K. and Smyth,  Gordon K. 2001. Tweedie family densities: methods of
    evaluation. In Proceedings of the 16th International Workshop on
    Statistical Modelling, Odense, Denmark, 2–6 July.

Jørgensen, B., Demétrio, C.G.B., Kristensen, E., Banta, G.T., Petersen, H.C.,
    Delefosse, M.: Bias-corrected Pearson estimating functions for Taylor’s
    power law applied to benthic macrofauna data. Stat. Probab. Lett. 81,
    749–758 (2011)

Smyth G.K. and Jørgensen B. 2002. Fitting Tweedie's compound Poisson model to
    insurance claims data: dispersion modelling. ASTIN Bulletin 32: 143–157
