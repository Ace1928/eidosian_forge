import dataclasses
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
class NonMaxSuppression(OpRun):

    def get_thresholds_from_inputs(self, pc: PrepareContext, max_output_boxes_per_class: int, iou_threshold: float, score_threshold: float) -> Tuple[int, float, float]:
        if pc.max_output_boxes_per_class_ is not None:
            max_output_boxes_per_class = max(pc.max_output_boxes_per_class_[0], 0)
        if pc.iou_threshold_ is not None:
            iou_threshold = pc.iou_threshold_[0]
        if pc.score_threshold_ is not None:
            score_threshold = pc.score_threshold_[0]
        return (max_output_boxes_per_class, iou_threshold, score_threshold)

    def prepare_compute(self, pc: PrepareContext, boxes_tensor: np.ndarray, scores_tensor: np.ndarray, max_output_boxes_per_class_tensor: np.ndarray, iou_threshold_tensor: np.ndarray, score_threshold_tensor: np.ndarray):
        pc.boxes_data_ = boxes_tensor
        pc.scores_data_ = scores_tensor
        if max_output_boxes_per_class_tensor.size != 0:
            pc.max_output_boxes_per_class_ = max_output_boxes_per_class_tensor
        if iou_threshold_tensor.size != 0:
            pc.iou_threshold_ = iou_threshold_tensor
        if score_threshold_tensor.size != 0:
            pc.score_threshold_ = score_threshold_tensor
        pc.boxes_size_ = boxes_tensor.size
        pc.scores_size_ = scores_tensor.size
        boxes_dims = boxes_tensor.shape
        scores_dims = scores_tensor.shape
        pc.num_batches_ = boxes_dims[0]
        pc.num_classes_ = scores_dims[1]
        pc.num_boxes_ = boxes_dims[1]

    def _run(self, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box):
        center_point_box = center_point_box or self.center_point_box
        pc = PrepareContext()
        self.prepare_compute(pc, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
        max_output_boxes_per_class, iou_threshold, score_threshold = self.get_thresholds_from_inputs(pc, 0, 0, 0)
        if max_output_boxes_per_class.size == 0:
            return (np.empty((0,), dtype=np.int64),)
        boxes_data = pc.boxes_data_
        scores_data = pc.scores_data_
        selected_indices = []
        for batch_index in range(pc.num_batches_):
            for class_index in range(pc.num_classes_):
                box_score_offset = (batch_index, class_index)
                batch_boxes = boxes_data[batch_index]
                candidate_boxes = []
                class_scores = scores_data[box_score_offset]
                if pc.score_threshold_ is not None:
                    for box_index in range(pc.num_boxes_):
                        if class_scores[box_index] > score_threshold:
                            candidate_boxes.append(BoxInfo(class_scores[box_index], box_index))
                else:
                    for box_index in range(pc.num_boxes_):
                        candidate_boxes.append(BoxInfo(class_scores[box_index], box_index))
                sorted_boxes = sorted(candidate_boxes)
                selected_boxes_inside_class = []
                while len(sorted_boxes) > 0 and len(selected_boxes_inside_class) < max_output_boxes_per_class:
                    next_top_score = sorted_boxes[-1]
                    selected = True
                    for selected_index in selected_boxes_inside_class:
                        if suppress_by_iou(batch_boxes, next_top_score.idx_, selected_index.idx_, center_point_box, iou_threshold):
                            selected = False
                            break
                    if selected:
                        selected_boxes_inside_class.append(next_top_score)
                        selected_indices.append(SelectedIndex(batch_index, class_index, next_top_score.idx_))
                    sorted_boxes.pop()
        result = np.empty((len(selected_indices), 3), dtype=np.int64)
        for i in range(result.shape[0]):
            result[i, 0] = selected_indices[i].batch_index_
            result[i, 1] = selected_indices[i].class_index_
            result[i, 2] = selected_indices[i].box_index_
        return (result,)