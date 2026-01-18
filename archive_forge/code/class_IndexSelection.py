from dataclasses import dataclass
from typing import List, Dict, Any, NewType, Optional
@dataclass(frozen=True, eq=True)
class IndexSelection:
    """
    An IndexSelection represents the state of an Altair
    point selection (as constructed by alt.selection_point())
    when neither the fields nor encodings arguments are specified.

    The value field is a list of zero-based indices into the
    selected dataset.

    Note: These indices only apply to the input DataFrame
    for charts that do not include aggregations (e.g. a scatter chart).
    """
    name: str
    value: List[int]
    store: Store

    @staticmethod
    def from_vega(name: str, signal: Optional[Dict[str, dict]], store: Store):
        """
        Construct an IndexSelection from the raw Vega signal and dataset values.

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
        IndexSelection
        """
        if signal is None:
            indices = []
        else:
            points = signal.get('vlPoint', {}).get('or', [])
            indices = [p['_vgsid_'] - 1 for p in points]
        return IndexSelection(name=name, value=indices, store=store)