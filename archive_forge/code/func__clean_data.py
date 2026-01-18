from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING
from ..util.sampledata import package_csv
def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    df = df.copy()
    df['mfr'] = [x.split()[0] for x in df.name]
    df.loc[df.mfr == 'chevy', 'mfr'] = 'chevrolet'
    df.loc[df.mfr == 'chevroelt', 'mfr'] = 'chevrolet'
    df.loc[df.mfr == 'maxda', 'mfr'] = 'mazda'
    df.loc[df.mfr == 'mercedes-benz', 'mfr'] = 'mercedes'
    df.loc[df.mfr == 'toyouta', 'mfr'] = 'toyota'
    df.loc[df.mfr == 'vokswagen', 'mfr'] = 'volkswagen'
    df.loc[df.mfr == 'vw', 'mfr'] = 'volkswagen'
    ORIGINS = ['North America', 'Europe', 'Asia']
    df.origin = [ORIGINS[x - 1] for x in df.origin]
    return df