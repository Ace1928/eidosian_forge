from . import _catboost

    Parameters
    ----------
    model :
        CatBoost / CatBoostClassifier / CatBoostRanker / CatBoostRegressor model
        NOTE: uplift allways use RawFormulaVal
    features :
        must be a dict mapping strings (factor names) or integers (flat indexes) into pairs of floats (base and next values)
    