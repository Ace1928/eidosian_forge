from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
class ImageAPIPolicy(APIPolicyBase):

    def __init__(self, context, image, enforcer=None):
        """Image API policy module.

        :param context: The RequestContext
        :param image: The ImageProxy object in question, or a dict of image
                      properties if no image is yet created or needed for
                      authorization context.
        :param enforcer: The policy.Enforcer object to use for enforcement
                         operations. If not provided (or None), the default
                         enforcer will be selected.
        """
        self._image = image
        if not self.is_created:
            target = {'project_id': image.get('owner', context.project_id), 'owner': image.get('owner', context.project_id), 'visibility': image.get('visibility', 'private')}
        else:
            target = policy.ImageTarget(image)
        super(ImageAPIPolicy, self).__init__(context, target, enforcer)

    @property
    def is_created(self):
        """Signal whether the image actually exists or not.

        False if the image is only being proposed by a create operation,
        True if it has already been created.
        """
        return not isinstance(self._image, dict)

    def _enforce(self, rule_name):
        """Translate Forbidden->NotFound for images."""
        try:
            super(ImageAPIPolicy, self)._enforce(rule_name)
        except webob.exc.HTTPForbidden:
            if not self.is_created:
                raise
            if rule_name == 'get_image' or not self.check('get_image'):
                raise webob.exc.HTTPNotFound()
            raise

    def check(self, name, *args):
        try:
            return super(ImageAPIPolicy, self).check(name, *args)
        except webob.exc.HTTPNotFound:
            return False

    def _enforce_visibility(self, visibility):
        try:
            policy._enforce_image_visibility(self.enforcer, self._context, visibility, self._target)
        except exception.Forbidden as e:
            raise webob.exc.HTTPForbidden(explanation=str(e))

    def update_property(self, name, value=None):
        if name == 'visibility':
            self._enforce_visibility(value)
        self.modify_image()

    def update_locations(self):
        self._enforce('set_image_location')

    def delete_locations(self):
        self._enforce('delete_image_location')
        if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope):
            check_is_image_mutable(self._context, self._image)

    def get_image_location(self):
        self._enforce('get_image_location')

    def add_image(self):
        try:
            self._enforce('add_image')
        except webob.exc.HTTPForbidden:
            if self._target['owner'] != self._context.project_id:
                msg = _("You are not permitted to create images owned by '%s'" % self._target['owner'])
                raise webob.exc.HTTPForbidden(msg)
            else:
                raise
        if 'visibility' in self._target:
            self._enforce_visibility(self._target['visibility'])
        if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope):
            check_admin_or_same_owner(self._context, self._target)

    def get_image(self):
        self._enforce('get_image')

    def get_images(self):
        self._enforce('get_images')

    def delete_image(self):
        self._enforce('delete_image')
        if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope):
            check_is_image_mutable(self._context, self._image)

    def upload_image(self):
        self._enforce('upload_image')
        if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope):
            check_is_image_mutable(self._context, self._image)

    def download_image(self):
        self._enforce('download_image')

    def modify_image(self):
        self._enforce('modify_image')
        if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope):
            check_is_image_mutable(self._context, self._image)

    def deactivate_image(self):
        self._enforce('deactivate')
        if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope):
            check_is_image_mutable(self._context, self._image)

    def reactivate_image(self):
        self._enforce('reactivate')
        if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope):
            check_is_image_mutable(self._context, self._image)

    def copy_image(self):
        self._enforce('copy_image')