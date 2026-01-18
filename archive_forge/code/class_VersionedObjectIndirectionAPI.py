import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
class VersionedObjectIndirectionAPI(object, metaclass=abc.ABCMeta):

    def object_action(self, context, objinst, objmethod, args, kwargs):
        """Perform an action on a VersionedObject instance.

        When indirection_api is set on a VersionedObject (to a class
        implementing this interface), method calls on remotable methods
        will cause this to be executed to actually make the desired
        call. This often involves performing RPC.

        :param context: The context within which to perform the action
        :param objinst: The object instance on which to perform the action
        :param objmethod: The name of the action method to call
        :param args: The positional arguments to the action method
        :param kwargs: The keyword arguments to the action method
        :returns: The result of the action method
        """
        pass

    def object_class_action(self, context, objname, objmethod, objver, args, kwargs):
        """.. deprecated:: 0.10.0

        Use :func:`object_class_action_versions` instead.

        Perform an action on a VersionedObject class.

        When indirection_api is set on a VersionedObject (to a class
        implementing this interface), classmethod calls on
        remotable_classmethod methods will cause this to be executed to
        actually make the desired call. This usually involves performing
        RPC.

        :param context: The context within which to perform the action
        :param objname: The registry name of the object
        :param objmethod: The name of the action method to call
        :param objver: The (remote) version of the object on which the
                       action is being taken
        :param args: The positional arguments to the action method
        :param kwargs: The keyword arguments to the action method
        :returns: The result of the action method, which may (or may not)
                  be an instance of the implementing VersionedObject class.
        """
        pass

    def object_class_action_versions(self, context, objname, objmethod, object_versions, args, kwargs):
        """Perform an action on a VersionedObject class.

        When indirection_api is set on a VersionedObject (to a class
        implementing this interface), classmethod calls on
        remotable_classmethod methods will cause this to be executed to
        actually make the desired call. This usually involves performing
        RPC.

        This differs from object_class_action() in that it is provided
        with object_versions, a manifest of client-side object versions
        for easier nested backports. The manifest is the result of
        calling obj_tree_get_versions().

        NOTE: This was not in the initial spec for this interface, so the
        base class raises NotImplementedError if you don't implement it.
        For backports, this method will be tried first, and if unimplemented,
        will fall back to object_class_action(). New implementations should
        provide this method instead of object_class_action()

        :param context: The context within which to perform the action
        :param objname: The registry name of the object
        :param objmethod: The name of the action method to call
        :param object_versions: A dict of {objname: version} mappings
        :param args: The positional arguments to the action method
        :param kwargs: The keyword arguments to the action method
        :returns: The result of the action method, which may (or may not)
                  be an instance of the implementing VersionedObject class.
        """
        warnings.warn('object_class_action() is deprecated in favor of object_class_action_versions() and will be removed in a later release', DeprecationWarning)
        raise NotImplementedError('Multi-version class action not supported')

    def object_backport(self, context, objinst, target_version):
        """.. deprecated:: 0.10.0

        Use :func:`object_backport_versions` instead.

        Perform a backport of an object instance to a specified version.

        When indirection_api is set on a VersionedObject (to a class
        implementing this interface), the default behavior of the base
        VersionedObjectSerializer, upon receiving an object with a version
        newer than what is in the lcoal registry, is to call this method to
        request a backport of the object. In an environment where there is
        an RPC-able service on the bus which can gracefully downgrade newer
        objects for older services, this method services as a translation
        mechanism for older code when receiving objects from newer code.

        NOTE: This older/original method is soon to be deprecated. When a
        backport is required, the newer object_backport_versions() will be
        tried, and if it raises NotImplementedError, then we will fall back
        to this (less optimal) method.

        :param context: The context within which to perform the backport
        :param objinst: An instance of a VersionedObject to be backported
        :param target_version: The maximum version of the objinst's class
                               that is understood by the requesting host.
        :returns: The downgraded instance of objinst
        """
        pass

    def object_backport_versions(self, context, objinst, object_versions):
        """Perform a backport of an object instance.

        This method is basically just like object_backport() but instead of
        providing a specific target version for the toplevel object and
        relying on the service-side mapping to handle sub-objects, this sends
        a mapping of all the dependent objects and their client-supported
        versions. The server will backport objects within the tree starting
        at objinst to the versions specified in object_versions, removing
        objects that have no entry. Use obj_tree_get_versions() to generate
        this mapping.

        NOTE: This was not in the initial spec for this interface, so the
        base class raises NotImplementedError if you don't implement it.
        For backports, this method will be tried first, and if unimplemented,
        will fall back to object_backport().

        :param context: The context within which to perform the backport
        :param objinst: An instance of a VersionedObject to be backported
        :param object_versions: A dict of {objname: version} mappings
        """
        warnings.warn('object_backport() is deprecated in favor of object_backport_versions() and will be removed in a later release', DeprecationWarning)
        raise NotImplementedError('Multi-version backport not supported')