import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
class TestObjectVersionChecker(test.TestCase):

    def setUp(self):
        super(TestObjectVersionChecker, self).setUp()
        objects = [MyObject, MyObject2]
        self.obj_classes = {obj.__name__: [obj] for obj in objects}
        self.ovc = fixture.ObjectVersionChecker(obj_classes=self.obj_classes)

    def test_get_hashes(self):
        fp = 'ashketchum'
        with mock.patch.object(self.ovc, '_get_fingerprint') as mock_gf:
            mock_gf.return_value = fp
            actual = self.ovc.get_hashes()
        expected = self._generate_hashes(self.obj_classes, fp)
        self.assertEqual(expected, actual, 'ObjectVersionChecker is not getting the fingerprints of all registered objects.')

    def test_get_hashes_with_extra_data(self):
        fp = 'garyoak'
        mock_func = mock.MagicMock()
        with mock.patch.object(self.ovc, '_get_fingerprint') as mock_gf:
            mock_gf.return_value = fp
            actual = self.ovc.get_hashes(extra_data_func=mock_func)
        expected = self._generate_hashes(self.obj_classes, fp)
        expected_calls = [((name,), {'extra_data_func': mock_func}) for name in self.obj_classes.keys()]
        self.assertEqual(expected, actual, 'ObjectVersionChecker is not getting the fingerprints of all registered objects.')
        self.assertEqual(len(expected_calls), len(mock_gf.call_args_list), 'get_hashes() did not call get the fingerprints of all objects in the registry.')
        for call in expected_calls:
            self.assertIn(call, mock_gf.call_args_list, 'get_hashes() did not call _get_fingerprint()correctly.')

    def test_test_hashes_none_changed(self):
        fp = 'pikachu'
        hashes = self._generate_hashes(self.obj_classes, fp)
        with mock.patch.object(self.ovc, 'get_hashes') as mock_gh:
            mock_gh.return_value = hashes
            actual_expected, actual_actual = self.ovc.test_hashes(hashes)
        expected_expected = expected_actual = {}
        self.assertEqual(expected_expected, actual_expected, "There are no objects changed, so the 'expected' return value should contain no objects.")
        self.assertEqual(expected_actual, actual_actual, "There are no objects changed, so the 'actual' return value should contain no objects.")

    def test_test_hashes_class_not_added(self):
        fp = 'gyrados'
        new_classes = copy.copy(self.obj_classes)
        self._add_class(new_classes, MyExtraObject)
        expected_hashes = self._generate_hashes(self.obj_classes, fp)
        actual_hashes = self._generate_hashes(new_classes, fp)
        with mock.patch.object(self.ovc, 'get_hashes') as mock_gh:
            mock_gh.return_value = actual_hashes
            actual_exp, actual_act = self.ovc.test_hashes(expected_hashes)
        expected_expected = {MyExtraObject.__name__: None}
        expected_actual = {MyExtraObject.__name__: fp}
        self.assertEqual(expected_expected, actual_exp, 'Expected hashes should not contain the fingerprint of the class that has not been added to the expected hash dictionary.')
        self.assertEqual(expected_actual, actual_act, 'The actual hash should contain the class that was added to the registry.')

    def test_test_hashes_new_fp_incorrect(self):
        fp1 = 'beedrill'
        fp2 = 'snorlax'
        expected_hashes = self._generate_hashes(self.obj_classes, fp1)
        actual_hashes = copy.copy(expected_hashes)
        actual_hashes[MyObject.__name__] = fp2
        with mock.patch.object(self.ovc, 'get_hashes') as mock_gh:
            mock_gh.return_value = actual_hashes
            actual_exp, actual_act = self.ovc.test_hashes(expected_hashes)
        expected_expected = {MyObject.__name__: fp1}
        expected_actual = {MyObject.__name__: fp2}
        self.assertEqual(expected_expected, actual_exp, 'Expected hashes should contain the updated object with the old hash.')
        self.assertEqual(expected_actual, actual_act, 'Actual hashes should contain the updated object with the new hash.')

    def test_test_hashes_passes_extra_func(self):
        mock_extra_func = mock.Mock()
        with mock.patch.object(self.ovc, 'get_hashes') as mock_get_hashes:
            self.ovc.test_hashes({}, extra_data_func=mock_extra_func)
            mock_get_hashes.assert_called_once_with(extra_data_func=mock_extra_func)

    def test_get_dependency_tree(self):
        with mock.patch.object(self.ovc, '_get_dependencies') as mock_gd:
            self.ovc.get_dependency_tree()
        expected_calls = [(({}, MyObject),), (({}, MyObject2),)]
        self.assertEqual(2, len(mock_gd.call_args_list), 'get_dependency_tree() tried to get the dependencies too many times.')
        for call in expected_calls:
            self.assertIn(call, mock_gd.call_args_list, 'get_dependency_tree() did not get the dependencies of the objects correctly.')

    def test_test_relationships_none_changed(self):
        dep_tree = {}
        self._add_dependency(MyObject, MyObject2, dep_tree)
        with mock.patch.object(self.ovc, 'get_dependency_tree') as mock_gdt:
            mock_gdt.return_value = dep_tree
            actual_exp, actual_act = self.ovc.test_relationships(dep_tree)
        expected_expected = expected_actual = {}
        self.assertEqual(expected_expected, actual_exp, "There are no objects changed, so the 'expected' return value should contain no objects.")
        self.assertEqual(expected_actual, actual_act, "There are no objects changed, so the 'actual' return value should contain no objects.")

    def test_test_relationships_rel_added(self):
        exp_tree = {}
        actual_tree = {}
        self._add_dependency(MyObject, MyObject2, exp_tree)
        self._add_dependency(MyObject, MyObject2, actual_tree)
        self._add_dependency(MyObject, MyExtraObject, actual_tree)
        with mock.patch.object(self.ovc, 'get_dependency_tree') as mock_gdt:
            mock_gdt.return_value = actual_tree
            actual_exp, actual_act = self.ovc.test_relationships(exp_tree)
        expected_expected = {'MyObject': {'MyObject2': '1.0'}}
        expected_actual = {'MyObject': {'MyObject2': '1.0', 'MyExtraObject': '1.0'}}
        self.assertEqual(expected_expected, actual_exp, 'The expected relationship tree is not being built from changes correctly.')
        self.assertEqual(expected_actual, actual_act, 'The actual relationship tree is not being built from changes correctly.')

    def test_test_relationships_class_added(self):
        exp_tree = {}
        actual_tree = {}
        self._add_dependency(MyObject, MyObject2, exp_tree)
        self._add_dependency(MyObject, MyObject2, actual_tree)
        self._add_dependency(MyObject2, MyExtraObject, actual_tree)
        with mock.patch.object(self.ovc, 'get_dependency_tree') as mock_gdt:
            mock_gdt.return_value = actual_tree
            actual_exp, actual_act = self.ovc.test_relationships(exp_tree)
        expected_expected = {'MyObject2': None}
        expected_actual = {'MyObject2': {'MyExtraObject': '1.0'}}
        self.assertEqual(expected_expected, actual_exp, 'The expected relationship tree is not being built from changes correctly.')
        self.assertEqual(expected_actual, actual_act, 'The actual relationship tree is not being built from changes correctly.')

    def test_test_compatibility_routines(self):
        del self.ovc.obj_classes[MyObject2.__name__]
        with mock.patch.object(self.ovc, '_test_object_compatibility') as toc:
            self.ovc.test_compatibility_routines()
        toc.assert_called_once_with(MyObject, manifest=None, init_args=[], init_kwargs={})

    def test_test_compatibility_routines_with_manifest(self):
        del self.ovc.obj_classes[MyObject2.__name__]
        man = {'who': 'cares'}
        with mock.patch.object(self.ovc, '_test_object_compatibility') as toc:
            with mock.patch('oslo_versionedobjects.base.obj_tree_get_versions') as otgv:
                otgv.return_value = man
                self.ovc.test_compatibility_routines(use_manifest=True)
        otgv.assert_called_once_with(MyObject.__name__)
        toc.assert_called_once_with(MyObject, manifest=man, init_args=[], init_kwargs={})

    def test_test_compatibility_routines_with_args_kwargs(self):
        del self.ovc.obj_classes[MyObject2.__name__]
        init_args = {MyObject: [1]}
        init_kwargs = {MyObject: {'foo': 'bar'}}
        with mock.patch.object(self.ovc, '_test_object_compatibility') as toc:
            self.ovc.test_compatibility_routines(init_args=init_args, init_kwargs=init_kwargs)
        toc.assert_called_once_with(MyObject, manifest=None, init_args=[1], init_kwargs={'foo': 'bar'})

    def test_test_relationships_in_order(self):
        with mock.patch.object(self.ovc, '_test_relationships_in_order') as mock_tr:
            self.ovc.test_relationships_in_order()
        expected_calls = [((MyObject,),), ((MyObject2,),)]
        self.assertEqual(2, len(mock_tr.call_args_list), 'test_relationships_in_order() tested too many relationships.')
        for call in expected_calls:
            self.assertIn(call, mock_tr.call_args_list, 'test_relationships_in_order() did not test the relationships of the individual objects correctly.')

    def test_test_relationships_in_order_positive(self):
        rels = {'bellsprout': [('1.0', '1.0'), ('1.1', '1.2'), ('1.3', '1.3')]}
        MyObject.obj_relationships = rels
        self.ovc._test_relationships_in_order(MyObject)

    def test_test_relationships_in_order_negative(self):
        rels = {'rattata': [('1.0', '1.0'), ('1.1', '1.2'), ('1.3', '1.1')]}
        MyObject.obj_relationships = rels
        self.assertRaises(AssertionError, self.ovc._test_relationships_in_order, MyObject)

    def test_find_remotable_method(self):
        method = self.ovc._find_remotable_method(MyObject, MyObject.remotable_method)
        self.assertEqual(MyObject.remotable_method.original_fn, method, '_find_remotable_method() did not find the remotable method of MyObject.')

    def test_find_remotable_method_classmethod(self):
        rcm = MyObject.remotable_classmethod
        method = self.ovc._find_remotable_method(MyObject, rcm)
        expected = rcm.__get__(None, MyObject).original_fn
        self.assertEqual(expected, method, '_find_remotable_method() did not find the remotable classmethod.')

    def test_find_remotable_method_non_remotable_method(self):
        nrm = MyObject.non_remotable_method
        method = self.ovc._find_remotable_method(MyObject, nrm)
        self.assertIsNone(method, "_find_remotable_method() found a method that isn't remotable.")

    def test_find_remotable_method_non_remotable_classmethod(self):
        nrcm = MyObject.non_remotable_classmethod
        method = self.ovc._find_remotable_method(MyObject, nrcm)
        self.assertIsNone(method, "_find_remotable_method() found a method that isn't remotable.")

    def test_get_fingerprint(self):
        MyObject.VERSION = '1.1'
        argspec = 'vulpix'
        with mock.patch.object(fixture, 'get_method_spec') as mock_gas:
            mock_gas.return_value = argspec
            fp = self.ovc._get_fingerprint(MyObject.__name__)
        exp_fields = sorted(list(MyObject.fields.items()))
        exp_methods = sorted([('remotable_method', argspec), ('remotable_classmethod', argspec)])
        expected_relevant_data = (exp_fields, exp_methods)
        expected_hash = hashlib.md5(bytes(repr(expected_relevant_data).encode())).hexdigest()
        expected_fp = '%s-%s' % (MyObject.VERSION, expected_hash)
        self.assertEqual(expected_fp, fp, '_get_fingerprint() did not generate a correct fingerprint.')

    def test_get_fingerprint_with_child_versions(self):
        child_versions = {'1.0': '1.0', '1.1': '1.1'}
        MyObject.VERSION = '1.1'
        MyObject.child_versions = child_versions
        argspec = 'onix'
        with mock.patch.object(fixture, 'get_method_spec') as mock_gas:
            mock_gas.return_value = argspec
            fp = self.ovc._get_fingerprint(MyObject.__name__)
        exp_fields = sorted(list(MyObject.fields.items()))
        exp_methods = sorted([('remotable_method', argspec), ('remotable_classmethod', argspec)])
        exp_child_versions = collections.OrderedDict(sorted(child_versions.items()))
        exp_relevant_data = (exp_fields, exp_methods, exp_child_versions)
        expected_hash = hashlib.md5(bytes(repr(exp_relevant_data).encode())).hexdigest()
        expected_fp = '%s-%s' % (MyObject.VERSION, expected_hash)
        self.assertEqual(expected_fp, fp, '_get_fingerprint() did not generate a correct fingerprint.')

    def test_get_fingerprint_with_extra_data(self):

        class ExtraDataObj(base.VersionedObject):
            pass

        def get_data(obj_class):
            return (obj_class,)
        ExtraDataObj.VERSION = '1.1'
        argspec = 'cubone'
        self._add_class(self.obj_classes, ExtraDataObj)
        with mock.patch.object(fixture, 'get_method_spec') as mock_gas:
            mock_gas.return_value = argspec
            fp = self.ovc._get_fingerprint(ExtraDataObj.__name__, extra_data_func=get_data)
        exp_fields = []
        exp_methods = []
        exp_extra_data = ExtraDataObj
        exp_relevant_data = (exp_fields, exp_methods, exp_extra_data)
        expected_hash = hashlib.md5(bytes(repr(exp_relevant_data).encode())).hexdigest()
        expected_fp = '%s-%s' % (ExtraDataObj.VERSION, expected_hash)
        self.assertEqual(expected_fp, fp, '_get_fingerprint() did not generate a correct fingerprint.')

    def test_get_fingerprint_with_defaulted_set(self):

        class ClassWithDefaultedSetField(base.VersionedObject):
            VERSION = 1.0
            fields = {'empty_default': fields.SetOfIntegersField(default=set()), 'non_empty_default': fields.SetOfIntegersField(default={1, 2})}
        self._add_class(self.obj_classes, ClassWithDefaultedSetField)
        expected = '1.0-bcc44920f2f727eca463c6eb4fe8445b'
        actual = self.ovc._get_fingerprint(ClassWithDefaultedSetField.__name__)
        self.assertEqual(expected, actual)

    def test_get_dependencies(self):
        self._add_class(self.obj_classes, MyExtraObject)
        MyObject.fields['subob'] = fields.ObjectField('MyExtraObject')
        MyExtraObject.VERSION = '1.0'
        tree = {}
        self.ovc._get_dependencies(tree, MyObject)
        expected_tree = {'MyObject': {'MyExtraObject': '1.0'}}
        self.assertEqual(expected_tree, tree, '_get_dependencies() did not generate a correct dependency tree.')

    def test_test_object_compatibility(self):
        to_prim = mock.MagicMock(spec=callable)
        MyObject.VERSION = '1.1'
        MyObject.obj_to_primitive = to_prim
        self.ovc._test_object_compatibility(MyObject)
        expected_calls = [((), {'target_version': '1.0'}), ((), {'target_version': '1.1'})]
        self.assertEqual(expected_calls, to_prim.call_args_list, '_test_object_compatibility() did not test obj_to_primitive() on the correct target versions')

    def test_test_object_compatibility_args_kwargs(self):
        to_prim = mock.MagicMock(spec=callable)
        MyObject.obj_to_primitive = to_prim
        MyObject.VERSION = '1.1'
        args = [1]
        kwargs = {'foo': 'bar'}
        with mock.patch.object(MyObject, '__init__', return_value=None) as mock_init:
            self.ovc._test_object_compatibility(MyObject, init_args=args, init_kwargs=kwargs)
        expected_init = ((1,), {'foo': 'bar'})
        expected_init_calls = [expected_init, expected_init]
        self.assertEqual(expected_init_calls, mock_init.call_args_list, '_test_object_compatibility() did not call __init__() properly on the object')
        expected_to_prim = [((), {'target_version': '1.0'}), ((), {'target_version': '1.1'})]
        self.assertEqual(expected_to_prim, to_prim.call_args_list, '_test_object_compatibility() did not test obj_to_primitive() on the correct target versions')

    def _add_class(self, obj_classes, cls):
        obj_classes[cls.__name__] = [cls]

    def _generate_hashes(self, classes, fp):
        return {cls: fp for cls in classes.keys()}

    def _add_dependency(self, parent_cls, child_cls, tree):
        deps = tree.get(parent_cls.__name__, {})
        deps[child_cls.__name__] = '1.0'
        tree[parent_cls.__name__] = deps