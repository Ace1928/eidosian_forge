import sys
def _good_enough(dom, features):
    """_good_enough(dom, features) -> Return 1 if the dom offers the features"""
    for f, v in features:
        if not dom.hasFeature(f, v):
            return 0
    return 1