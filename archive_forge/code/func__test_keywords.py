from ase import Atom, Atoms
def _test_keywords(**kw):
    was_raised = False
    try:
        Atoms(**kw)
    except Exception as inst:
        assert isinstance(inst, TypeError), inst
        was_raised = True
    assert was_raised