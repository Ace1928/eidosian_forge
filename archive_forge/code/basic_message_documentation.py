from .serialization import GenericContent
from .spec import Basic
A Message for use with the Channel.basic_* methods.

    Expected arg types

        body: string
        children: (not supported)

    Keyword properties may include:

        content_type: shortstr
            MIME content type

        content_encoding: shortstr
            MIME content encoding

        application_headers: table
            Message header field table, a dict with string keys,
            and string | int | Decimal | datetime | dict values.

        delivery_mode: octet
            Non-persistent (1) or persistent (2)

        priority: octet
            The message priority, 0 to 9

        correlation_id: shortstr
            The application correlation identifier

        reply_to: shortstr
            The destination to reply to

        expiration: shortstr
            Message expiration specification

        message_id: shortstr
            The application message identifier

        timestamp: unsigned long
            The message timestamp

        type: shortstr
            The message type name

        user_id: shortstr
            The creating user id

        app_id: shortstr
            The creating application id

        cluster_id: shortstr
            Intra-cluster routing identifier

        Unicode bodies are encoded according to the 'content_encoding'
        argument. If that's None, it's set to 'UTF-8' automatically.

        Example::

            msg = Message('hello world',
                            content_type='text/plain',
                            application_headers={'foo': 7})
    