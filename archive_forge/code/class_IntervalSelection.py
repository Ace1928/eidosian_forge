from dataclasses import dataclass
from typing import List, Dict, Any, NewType, Optional
@dataclass(frozen=True, eq=True)
class IntervalSelection:
    """
    An IntervalSelection represents the state of an Altair
    interval selection (as constructed by alt.selection_interval()).

    The value field is a dict of the form:
        {"dim1": [0, 10], "dim2": ["A", "BB", "CCC"]}

    where "dim1" and "dim2" are dataset columns and the dict values
    correspond to the selected range.
    """
    name: str
    value: Dict[str, list]
    store: Store

    @staticmethod
    def from_vega(name: str, signal: Optional[Dict[str, list]], store: Store):
        """
        Construct an IntervalSelection from the raw Vega signal and dataset values.

        Parameters
        ----------
        name: str
            The selection's name
        signal: dict or None
            The value of the Vega signal corresponding to the selection
        store: list
            The value of the Vega dataset corresponding to the selection.
            This dataset is named "{name}_store" in the Vega view.

        Returns
        -------
        PointSelection
        """
        if signal is None:
            signal = {}
        return IntervalSelection(name=name, value=signal, store=store)