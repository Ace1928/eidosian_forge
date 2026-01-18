from passlib.registry import register_crypt_handler
import passlib.utils.handlers as uh
class dummy_bad(uh.StaticHandler):
    name = 'dummy_bad'