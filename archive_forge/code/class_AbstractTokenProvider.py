import abc
class AbstractTokenProvider(abc.ABC):
    """
    A Token Provider must be used for the `SASL OAuthBearer`_ protocol.

    The implementation should ensure token reuse so that multiple
    calls at connect time do not create multiple tokens.
    The implementation should also periodically refresh the token in order to
    guarantee that each call returns an unexpired token.

    A timeout error should be returned after a short period of inactivity so
    that the broker can log debugging info and retry.

    Token Providers MUST implement the :meth:`token` method

    .. _SASL OAuthBearer:
        https://docs.confluent.io/platform/current/kafka/authentication_sasl/authentication_sasl_oauth.html
    """

    def __init__(self, **config):
        pass

    @abc.abstractmethod
    async def token(self):
        """
        An async callback returning a :class:`str` ID/Access Token to be sent to
        the Kafka client. In case where a synchoronous callback is needed,
        implementations like following can be used:

        .. code-block:: python

            from aiokafka.abc import AbstractTokenProvider

            class CustomTokenProvider(AbstractTokenProvider):
                async def token(self):
                    return await asyncio.get_running_loop().run_in_executor(
                        None, self._token)

                def _token(self):
                    # The actual synchoronous token callback.
        """
        pass

    def extensions(self):
        """
        This is an OPTIONAL method that may be implemented.

        Returns a map of key-value pairs that can be sent with the
        SASL/OAUTHBEARER initial client request. If not implemented, the values
        are ignored

        This feature is only available in Kafka >= 2.1.0.
        """
        return {}