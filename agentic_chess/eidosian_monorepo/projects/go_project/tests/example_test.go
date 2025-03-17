package main

import "testing"

func TestExample(t *testing.T) {
    if 1 != 1 {
        t.Errorf("Expected 1 to equal 1")
    }
}
