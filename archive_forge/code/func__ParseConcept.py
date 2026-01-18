from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
import six
def _ParseConcept(node):
    """Recursive parsing."""
    if not node.is_group:
        fallthroughs = []
        if node.arg_name:
            fallthroughs.append(deps_lib.ArgFallthrough(node.arg_name))
        fallthroughs += node.fallthroughs
        return node.concept.Parse(DependencyViewFromValue(functools.partial(deps_lib.GetFromFallthroughs, fallthroughs, parsed_args), marshalled_dependencies=node.dependencies))
    also_optional = []
    have_optional = []
    have_required = []
    need_required = []
    namespace = {}
    for name, child in six.iteritems(node.dependencies):
        result = None
        try:
            result = _ParseConcept(child)
            if result:
                if child.concept.required:
                    have_required.append(child.concept)
                else:
                    have_optional.append(child.concept)
            else:
                also_optional.append(child.concept)
        except exceptions.MissingRequiredArgumentError:
            need_required.append(child.concept)
        namespace[name] = result
    if need_required:
        missing = ' '.join(GetPresentationNames(need_required))
        if have_optional or have_required:
            specified_parts = []
            if have_required:
                specified_parts.append(' '.join(GetPresentationNames(have_required)))
            if have_required and have_optional:
                specified_parts.append(':')
            if have_optional:
                specified_parts.append(' '.join(GetPresentationNames(have_optional)))
            specified = ' '.join(specified_parts)
            if have_required and have_optional:
                if node.concept.required:
                    specified = '({})'.format(specified)
                else:
                    specified = '[{}]'.format(specified)
            raise exceptions.ModalGroupError(node.concept.GetPresentationName(), specified, missing)
    count = len(have_required) + len(have_optional)
    if node.concept.mutex:
        specified = ' | '.join(GetPresentationNames(node.concept.concepts))
        if node.concept.required:
            specified = '({specified})'.format(specified=specified)
            if count != 1:
                raise exceptions.RequiredMutexGroupError(node.concept.GetPresentationName(), specified)
        elif count > 1:
            raise exceptions.OptionalMutexGroupError(node.concept.GetPresentationName(), specified)
    return node.concept.Parse(DependencyView(namespace))