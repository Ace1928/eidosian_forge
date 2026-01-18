from llvmlite import binding as llvm
def _inlining_threshold(optlevel, sizelevel=0):
    """
    Compute the inlining threshold for the desired optimisation level

    Refer to http://llvm.org/docs/doxygen/html/InlineSimple_8cpp_source.html
    """
    if optlevel > 2:
        return 275
    if sizelevel == 1:
        return 75
    if sizelevel == 2:
        return 25
    return 225