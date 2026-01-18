import os
from ..base import (
from ...external.due import BibTeX
from .base import (
class RemlfitOutputSpec(AFNICommandOutputSpec):
    out_file = File(desc='dataset for beta + statistics from the REML estimation (if generated')
    var_file = File(desc='dataset for REML variance parameters (if generated)')
    rbeta_file = File(desc='dataset for beta weights from the REML estimation (if generated)')
    rbeta_file = File(desc='output dataset for beta weights from the REML estimation (if generated')
    glt_file = File(desc="output dataset for beta + statistics from the REML estimation, but ONLY for the GLTs added on the REMLfit command line itself via 'gltsym' (if generated)")
    fitts_file = File(desc='output dataset for REML fitted model (if generated)')
    errts_file = File(desc='output dataset for REML residuals = data - fitted model (if generated')
    wherr_file = File(desc='dataset for REML residual, whitened using the estimated ARMA(1,1) correlation matrix of the noise (if generated)')
    ovar = File(desc='dataset for OLSQ st.dev. parameter (if generated)')
    obeta = File(desc='dataset for beta weights from the OLSQ estimation (if generated)')
    obuck = File(desc='dataset for beta + statistics from the OLSQ estimation (if generated)')
    oglt = File(desc="dataset for beta + statistics from 'gltsym' options (if generated")
    ofitts = File(desc='dataset for OLSQ fitted model (if generated)')
    oerrts = File(desc='dataset for OLSQ residuals = data - fitted model (if generated')