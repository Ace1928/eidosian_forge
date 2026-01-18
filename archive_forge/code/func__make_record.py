import logging
import time
from abc import abstractmethod
from multiprocessing.process import BaseProcess
from typing import Any, Optional, cast
import wandb
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from wandb.util import json_dumps_safer, json_friendly
from ..lib.mailbox import Mailbox, MailboxHandle
from .interface import InterfaceBase
from .message_future import MessageFuture
from .router import MessageRouter
def _make_record(self, run: Optional[pb.RunRecord]=None, config: Optional[pb.ConfigRecord]=None, files: Optional[pb.FilesRecord]=None, summary: Optional[pb.SummaryRecord]=None, history: Optional[pb.HistoryRecord]=None, stats: Optional[pb.StatsRecord]=None, exit: Optional[pb.RunExitRecord]=None, artifact: Optional[pb.ArtifactRecord]=None, tbrecord: Optional[pb.TBRecord]=None, alert: Optional[pb.AlertRecord]=None, final: Optional[pb.FinalRecord]=None, metric: Optional[pb.MetricRecord]=None, header: Optional[pb.HeaderRecord]=None, footer: Optional[pb.FooterRecord]=None, request: Optional[pb.Request]=None, telemetry: Optional[tpb.TelemetryRecord]=None, preempting: Optional[pb.RunPreemptingRecord]=None, link_artifact: Optional[pb.LinkArtifactRecord]=None, use_artifact: Optional[pb.UseArtifactRecord]=None, output: Optional[pb.OutputRecord]=None, output_raw: Optional[pb.OutputRawRecord]=None) -> pb.Record:
    record = pb.Record()
    if run:
        record.run.CopyFrom(run)
    elif config:
        record.config.CopyFrom(config)
    elif summary:
        record.summary.CopyFrom(summary)
    elif history:
        record.history.CopyFrom(history)
    elif files:
        record.files.CopyFrom(files)
    elif stats:
        record.stats.CopyFrom(stats)
    elif exit:
        record.exit.CopyFrom(exit)
    elif artifact:
        record.artifact.CopyFrom(artifact)
    elif tbrecord:
        record.tbrecord.CopyFrom(tbrecord)
    elif alert:
        record.alert.CopyFrom(alert)
    elif final:
        record.final.CopyFrom(final)
    elif header:
        record.header.CopyFrom(header)
    elif footer:
        record.footer.CopyFrom(footer)
    elif request:
        record.request.CopyFrom(request)
    elif telemetry:
        record.telemetry.CopyFrom(telemetry)
    elif metric:
        record.metric.CopyFrom(metric)
    elif preempting:
        record.preempting.CopyFrom(preempting)
    elif link_artifact:
        record.link_artifact.CopyFrom(link_artifact)
    elif use_artifact:
        record.use_artifact.CopyFrom(use_artifact)
    elif output:
        record.output.CopyFrom(output)
    elif output_raw:
        record.output_raw.CopyFrom(output_raw)
    else:
        raise Exception('Invalid record')
    return record