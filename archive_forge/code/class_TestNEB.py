from pytest import warns, raises
from ase import Atoms
from ase import neb
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
class TestNEB(object):

    @classmethod
    def setup_class(cls):
        cls.h_atom = Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0])
        cls.h2_molecule = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cls.images_dummy = [cls.h_atom.copy(), cls.h_atom.copy(), cls.h_atom.copy()]

    def test_deprecations(self, testdir):
        with warns(FutureWarning, match='.*Please use.*'):
            deprecated_neb = neb.SingleCalculatorNEB(self.images_dummy)
        assert deprecated_neb.allow_shared_calculator
        neb_dummy = neb.NEB(self.images_dummy)
        with warns(FutureWarning, match='.*Please use.*idpp_interpolate.*'):
            neb_dummy.idpp_interpolate(steps=1)

    def test_neb_default(self):
        neb_dummy = neb.NEB(self.images_dummy)
        assert not neb_dummy.allow_shared_calculator

    def test_raising_parallel_errors(self):
        with raises(RuntimeError, match='.*Cannot use shared calculators.*'):
            _ = neb.NEB(self.images_dummy, allow_shared_calculator=True, parallel=True)

    def test_no_shared_calc(self):
        images_shared_calc = [self.h_atom.copy(), self.h_atom.copy(), self.h_atom.copy()]
        shared_calc = EMT()
        for at in images_shared_calc:
            at.calc = shared_calc
        neb_not_allow = neb.NEB(images_shared_calc, allow_shared_calculator=False)
        with raises(ValueError, match='.*NEB images share the same.*'):
            neb_not_allow.get_forces()
        with raises(RuntimeError, match='.*Cannot set shared calculator.*'):
            neb_not_allow.set_calculators(EMT())
        new_calculators = [EMT() for _ in range(neb_not_allow.nimages)]
        neb_not_allow.set_calculators(new_calculators)
        for i in range(neb_not_allow.nimages):
            assert new_calculators[i] == neb_not_allow.images[i].calc
        neb_not_allow.set_calculators(new_calculators[1:-1])
        for i in range(1, neb_not_allow.nimages - 1):
            assert new_calculators[i] == neb_not_allow.images[i].calc
        with raises(RuntimeError, match='.*does not fit to len.*'):
            neb_not_allow.set_calculators(new_calculators[:-1])

    def test_init_checks(self):
        mismatch_len = [self.h_atom.copy(), self.h2_molecule.copy()]
        with raises(ValueError, match='.*different numbers of atoms.*'):
            _ = neb.NEB(mismatch_len)
        mismatch_pbc = [self.h_atom.copy(), self.h_atom.copy()]
        mismatch_pbc[-1].set_pbc(True)
        with raises(ValueError, match='.*different boundary conditions.*'):
            _ = neb.NEB(mismatch_pbc)
        mismatch_numbers = [self.h_atom.copy(), Atoms('C', positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0])]
        with raises(ValueError, match='.*atoms in different orders.*'):
            _ = neb.NEB(mismatch_numbers)
        mismatch_cell = [self.h_atom.copy(), self.h_atom.copy()]
        mismatch_cell[-1].set_cell(mismatch_cell[-1].get_cell() + 1e-05)
        with raises(NotImplementedError, match='.*Variable cell.*'):
            _ = neb.NEB(mismatch_cell)

    def test_freeze_method(self):
        at = self.h_atom.copy()
        at.calc = EMT()
        at.get_forces()
        results = dict(**at.calc.results)
        neb.NEB.freeze_results_on_image(at, **results)
        assert isinstance(at.calc, SinglePointCalculator)