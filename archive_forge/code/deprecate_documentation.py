from typing import TYPE_CHECKING, Optional, Tuple
import wandb
from wandb.proto.wandb_deprecated import DEPRECATED_FEATURES, Deprecated
from wandb.proto.wandb_telemetry_pb2 import Deprecated as TelemetryDeprecated
Warn the user that a feature has been deprecated.

    Also stores the information about the event in telemetry.

    Args:
        field_name: The name of the feature that has been deprecated.
                    Defined in wandb/proto/wandb_telemetry.proto::Deprecated
        warning_message: The message to display to the user.
        run: The run to whose telemetry the event will be added.
    