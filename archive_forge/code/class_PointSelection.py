from dataclasses import dataclass
from typing import List, Dict, Any, NewType, Optional
@dataclass(frozen=True, eq=True)
class PointSelection:
    """
    A PointSelection represents the state of an Altair
    point selection (as constructed by alt.selection_point())
    when the fields or encodings arguments are specified.

    The value field is a list of dicts of the form:
        [{"dim1": 1, "dim2": "A"}, {"dim1": 2, "dim2": "BB"}]

    where "dim1" and "dim2" are dataset columns and the dict values
    correspond to the specific selected values.
    """
    name: str
    value: List[Dict[str, Any]]
    store: Store

    @staticmethod
    def from_vega(name: str, signal: Optional[Dict[str, dict]], store: Store):
        """
        Construct a PointSelection from the raw Vega signal and dataset values.

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
            points = []
        else:
            points = signal.get('vlPoint', {}).get('or', [])
        return PointSelection(name=name, value=points, store=store)