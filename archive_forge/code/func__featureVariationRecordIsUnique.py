from fontTools.ttLib.tables import otTables as ot
from copy import deepcopy
import logging
def _featureVariationRecordIsUnique(rec, seen):
    conditionSet = []
    conditionSets = rec.ConditionSet.ConditionTable if rec.ConditionSet is not None else []
    for cond in conditionSets:
        if cond.Format != 1:
            return True
        conditionSet.append((cond.AxisIndex, cond.FilterRangeMinValue, cond.FilterRangeMaxValue))
    recordKey = frozenset([rec.FeatureTableSubstitution.Version] + conditionSet)
    if recordKey in seen:
        return False
    else:
        seen.add(recordKey)
        return True