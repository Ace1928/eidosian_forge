from fontTools.ttLib.tables import otTables as ot
from copy import deepcopy
import logging
def _instantiateFeatureVariationRecord(record, recIdx, axisLimits, fvarAxes, axisIndexMap):
    applies = True
    shouldKeep = False
    newConditions = []
    from fontTools.varLib.instancer import NormalizedAxisTripleAndDistances
    default_triple = NormalizedAxisTripleAndDistances(-1, 0, +1)
    if record.ConditionSet is None:
        record.ConditionSet = ot.ConditionSet()
        record.ConditionSet.ConditionTable = []
        record.ConditionSet.ConditionCount = 0
    for i, condition in enumerate(record.ConditionSet.ConditionTable):
        if condition.Format == 1:
            axisIdx = condition.AxisIndex
            axisTag = fvarAxes[axisIdx].axisTag
            minValue = condition.FilterRangeMinValue
            maxValue = condition.FilterRangeMaxValue
            triple = axisLimits.get(axisTag, default_triple)
            if not minValue <= triple.default <= maxValue:
                applies = False
            if triple.minimum > maxValue or triple.maximum < minValue:
                newConditions = None
                break
            if axisTag in axisIndexMap:
                condition.AxisIndex = axisIndexMap[axisTag]
                newRange = _limitFeatureVariationConditionRange(condition, triple)
                if newRange:
                    minimum, maximum = newRange
                    condition.FilterRangeMinValue = minimum
                    condition.FilterRangeMaxValue = maximum
                    shouldKeep = True
                    if minimum != -1 or maximum != +1:
                        newConditions.append(condition)
                else:
                    newConditions = None
                    break
        else:
            log.warning('Condition table {0} of FeatureVariationRecord {1} has unsupported format ({2}); ignored'.format(i, recIdx, condition.Format))
            applies = False
            newConditions.append(condition)
    if newConditions is not None and shouldKeep:
        record.ConditionSet.ConditionTable = newConditions
        if not newConditions:
            record.ConditionSet = None
        shouldKeep = True
    else:
        shouldKeep = False
    universal = shouldKeep and (not newConditions)
    return (applies, shouldKeep, universal)