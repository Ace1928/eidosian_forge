from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def SanitizePlatformSettings(unused_ref, args, request):
    """Make sure that at most one platform setting is set at the same time."""
    if args.android:
        request.googleCloudRecaptchaenterpriseV1Key.iosSettings = None
        request.googleCloudRecaptchaenterpriseV1Key.webSettings = None
    elif args.ios:
        request.googleCloudRecaptchaenterpriseV1Key.androidSettings = None
        request.googleCloudRecaptchaenterpriseV1Key.webSettings = None
    elif args.web:
        request.googleCloudRecaptchaenterpriseV1Key.androidSettings = None
        request.googleCloudRecaptchaenterpriseV1Key.iosSettings = None
    else:
        request.googleCloudRecaptchaenterpriseV1Key.androidSettings = None
        request.googleCloudRecaptchaenterpriseV1Key.iosSettings = None
        request.googleCloudRecaptchaenterpriseV1Key.webSettings = None
    return request