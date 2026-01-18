from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from .ack import AckProtocolEntity

    <ack t="{{TIMESTAMP}}" from="{{FROM_JID}}" id="{{MESSAGE_ID}}" class="{{message | receipt | ?}}">
    </ack>
    