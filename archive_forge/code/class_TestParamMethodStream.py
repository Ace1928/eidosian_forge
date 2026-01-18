from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
class TestParamMethodStream(ComparisonTestCase):

    def setUp(self):
        if Version(param.__version__) < Version('1.8.0'):
            raise SkipTest('Params stream requires param >= 1.8.0')

        class Inner(param.Parameterized):
            action = param.Action(default=lambda o: o.param.trigger('action'))
            x = param.Number(default=0)
            y = param.Number(default=0)
            count = param.Integer(default=0)

            @param.depends('x')
            def method(self):
                self.count += 1
                return Points([])

            @param.depends('action')
            def action_method(self):
                pass

            @param.depends('action', 'x')
            def action_number_method(self):
                self.count += 1
                return Points([])

            @param.depends('y')
            def op_method(self, obj):
                pass

            def method_no_deps(self):
                pass

        class InnerSubObj(Inner):
            sub = param.Parameter()

            @param.depends('sub.x')
            def subobj_method(self):
                pass
        self.inner = Inner
        self.innersubobj = InnerSubObj

    def test_param_method_depends(self):
        inner = self.inner()
        stream = ParamMethod(inner.method)
        self.assertEqual(set(stream.parameters), {inner.param.x})
        self.assertEqual(stream.contents, {})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.y = 2
        self.assertEqual(values, [{}])

    def test_param_function_depends(self):
        inner = self.inner()

        @param.depends(inner.param.x)
        def test(x):
            return Points([x])
        dmap = DynamicMap(test)
        inner.x = 10
        self.assertEqual(dmap[()], Points([10]))

    def test_param_instance_steams_dict(self):
        inner = self.inner()

        def test(x):
            return Points([x])
        dmap = DynamicMap(test, streams=dict(x=inner.param.x))
        inner.x = 10
        self.assertEqual(dmap[()], Points([10]))

    def test_param_class_steams_dict(self):

        class ClassParamExample(param.Parameterized):
            x = param.Number(default=1)

        def test(x):
            return Points([x])
        dmap = DynamicMap(test, streams=dict(x=ClassParamExample.param.x))
        ClassParamExample.x = 10
        self.assertEqual(dmap[()], Points([10]))

    def test_panel_param_steams_dict(self):
        import panel as pn
        widget = pn.widgets.FloatSlider(value=1)

        def test(x):
            return Points([x])
        dmap = DynamicMap(test, streams=dict(x=widget))
        widget.value = 10
        self.assertEqual(dmap[()], Points([10]))

    def test_param_method_depends_no_deps(self):
        inner = self.inner()
        stream = ParamMethod(inner.method_no_deps)
        self.assertEqual(set(stream.parameters), {inner.param.x, inner.param.y, inner.param.action, inner.param.name, inner.param.count})
        self.assertEqual(stream.contents, {})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.y = 2
        self.assertEqual(values, [{}, {}])

    def test_param_method_depends_on_subobj(self):
        inner = self.innersubobj(sub=self.inner())
        stream = ParamMethod(inner.subobj_method)
        self.assertEqual(set(stream.parameters), {inner.sub.param.x})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        inner.sub.x = 2
        self.assertEqual(values, [{}])

    def test_dynamicmap_param_method_deps(self):
        inner = self.inner()
        dmap = DynamicMap(inner.method)
        self.assertEqual(len(dmap.streams), 1)
        stream = dmap.streams[0]
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.y = 2
        self.assertEqual(values, [{}])

    def test_param_method_depends_trigger_no_memoization(self):
        inner = self.inner()
        stream = ParamMethod(inner.method)
        self.assertEqual(set(stream.parameters), {inner.param.x})
        self.assertEqual(stream.contents, {})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
        stream.add_subscriber(subscriber)
        inner.x = 2
        inner.param.trigger('x')
        self.assertEqual(values, [{}, {}])

    def test_dynamicmap_param_method_deps_memoization(self):
        inner = self.inner()
        dmap = DynamicMap(inner.method)
        stream = dmap.streams[0]
        self.assertEqual(set(stream.parameters), {inner.param.x})
        self.assertEqual(stream.contents, {})
        dmap[()]
        dmap[()]
        self.assertEqual(inner.count, 1)

    def test_dynamicmap_param_method_no_deps(self):
        inner = self.inner()
        dmap = DynamicMap(inner.method_no_deps)
        self.assertEqual(dmap.streams, [])

    def test_dynamicmap_param_method_action_param(self):
        inner = self.inner()
        dmap = DynamicMap(inner.action_method)
        self.assertEqual(len(dmap.streams), 1)
        stream = dmap.streams[0]
        self.assertEqual(set(stream.parameters), {inner.param.action})
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey), {f'{id(inner)} action', '_memoize_key'})
        stream.add_subscriber(subscriber)
        inner.action(inner)
        self.assertEqual(values, [{}])

    def test_dynamicmap_param_action_number_method_memoizes(self):
        inner = self.inner()
        dmap = DynamicMap(inner.action_number_method)
        self.assertEqual(len(dmap.streams), 1)
        stream = dmap.streams[0]
        self.assertEqual(set(stream.parameters), {inner.param.action, inner.param.x})
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})
        values = []

        def subscriber(**kwargs):
            values.append(kwargs)
            self.assertEqual(set(stream.hashkey), {f'{id(inner)} action', f'{id(inner)} x', '_memoize_key'})
        stream.add_subscriber(subscriber)
        stream.add_subscriber(lambda **kwargs: dmap[()])
        inner.action(inner)
        self.assertEqual(values, [{}])
        self.assertEqual(inner.count, 1)
        inner.x = 0
        self.assertEqual(values, [{}])
        self.assertEqual(inner.count, 1)

    def test_dynamicmap_param_method_dynamic_operation(self):
        inner = self.inner()
        dmap = DynamicMap(inner.method)
        inner_stream = dmap.streams[0]
        op_dmap = Dynamic(dmap, operation=inner.op_method)
        self.assertEqual(len(op_dmap.streams), 1)
        stream = op_dmap.streams[0]
        self.assertEqual(set(stream.parameters), {inner.param.y})
        self.assertIsInstance(stream, ParamMethod)
        self.assertEqual(stream.contents, {})
        values_x, values_y = ([], [])

        def subscriber_x(**kwargs):
            values_x.append(kwargs)

        def subscriber_y(**kwargs):
            values_y.append(kwargs)
        inner_stream.add_subscriber(subscriber_x)
        stream.add_subscriber(subscriber_y)
        inner.y = 3
        self.assertEqual(values_x, [])
        self.assertEqual(values_y, [{}])